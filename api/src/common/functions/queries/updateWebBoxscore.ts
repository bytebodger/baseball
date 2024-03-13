import { IdentifyField } from '../../enums/IdentifyField.js';
import { Table } from '../../enums/Table.js';
import { updateTableByPrimaryKey } from './updateTableByPrimaryKey.js';

interface Fields {
   html?: string,
   season?: number,
   time_processed?: number,
   time_retrieved?: number,
   url?: string,
   web_boxscore_id: number,
}

export const updateWebBoxscore = async (fields: Fields) => {
   return await updateTableByPrimaryKey(Table.webBoxScore, IdentifyField.webBoxScore, fields);
}