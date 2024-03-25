import { IdentityField } from '../../enums/IdentityField.js';
import { Table } from '../../enums/Table.js';
import { updateDBTableByPrimaryKey } from './updateDBTableByPrimaryKey.js';

interface Fields {
   html?: string,
   season?: number,
   time_processed?: number,
   time_retrieved?: number,
   url?: string,
   web_boxscore_id: number,
}

export const updateDBWebBoxscore = async (fields: Fields) => {
   return await updateDBTableByPrimaryKey(Table.webBoxScore, IdentityField.webBoxScore, fields);
}