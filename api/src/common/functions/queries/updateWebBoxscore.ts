import { IdentifyField } from '../../enums/IdentifyField.js';
import { Table } from '../../enums/Table.js';
import { updateByPrimaryKey } from './updateByPrimaryKey.js';

interface Columns {
   html?: string,
   season?: number,
   time_processed?: number,
   time_retrieved?: number,
   url?: string,
   web_boxscore_id: number,
}

export const updateWebBoxscore = async (columns: Columns) => {
   return await updateByPrimaryKey(Table.webBoxScore, IdentifyField.webBoxScore, columns);
}